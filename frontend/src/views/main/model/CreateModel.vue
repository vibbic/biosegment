<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Create Model</div>
      </v-card-title>
      <v-card-text>
        <ModelForm :model="newModel"></ModelForm>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit"> Save </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Model, ModelUpdate, ModelCreate } from '@/api';
import { defaultModel } from '@/interfaces';
import ModelForm from '@/components/ModelForm.vue';
import { dispatchCreateModel } from '@/store/model/actions';
import { filterUndefined } from '@/utils';

@Component({ components: { ModelForm } })
export default class CreateModel extends Vue {
  public newModel: ModelCreate = defaultModel();
  public valid = false;

  public async mounted() {
    this.reset();
  }

  public reset() {
    this.newModel = defaultModel();
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      const filteredModel: ModelCreate = filterUndefined(this.newModel);
      await dispatchCreateModel(this.$store, filteredModel);
      this.$router.push('/main/models');
    }
  }
}
</script>
