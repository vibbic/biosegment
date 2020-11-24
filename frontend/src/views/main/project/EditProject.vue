<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Edit Project</div>
      </v-card-title>
      <v-card-text>
        <ProjectForm
          :project="projectForm"
          title="Update Project"
        ></ProjectForm>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit">Save</v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Project, ProjectUpdate, ProjectCreate } from '@/api';
import { defaultProject } from '@/interfaces';
import ProjectForm from '@/components/ProjectForm.vue';
import {
  dispatchGetProjects,
  dispatchUpdateProject,
} from '@/store/project/actions';
import { component } from 'vue/types/umd';
import { readOneProject } from '@/store/project/getters';
import { filterUndefined, deepCopy } from '@/utils';

@Component({ components: { ProjectForm } })
export default class EditProject extends Vue {
  public projectForm: ProjectUpdate = deepCopy(this.project);
  public valid = false;

  public async mounted() {
    await dispatchGetProjects(this.$store);
    this.reset();
  }

  public reset() {
    this.projectForm = deepCopy(this.project);
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      const filteredProject: ProjectUpdate = filterUndefined(this.projectForm);
      await dispatchUpdateProject(this.$store, {
        id: this.project!.id,
        project: filteredProject,
      });
      this.$router.push('/main/projects');
    }
  }

  get project() {
    return readOneProject(this.$store)(+this.$router.currentRoute.params.id);
  }
}
</script>
