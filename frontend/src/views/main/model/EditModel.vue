<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Edit Model</div>
      </v-card-title>
      <v-card-text>
        <ModelForm
          :model="modelForm"
          title="Update Model"
        ></ModelForm>
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
import { Model, ModelUpdate, ModelCreate } from '@/api';
import { defaultModel } from '@/interfaces';
import ModelForm from '@/components/ModelForm.vue';
import {
  dispatchGetModels,
  dispatchUpdateModel,
} from '@/store/model/actions';
import { component } from 'vue/types/umd';
import { readOneModel } from '@/store/model/getters';
import { filterUndefined, deepCopy } from '@/utils';

@Component({ components: { ModelForm } })
export default class EditModel extends Vue {
  public modelForm: ModelUpdate = deepCopy(this.model);
  public valid = false;

  public async mounted() {
    await dispatchGetModels(this.$store);
    this.reset();
  }

  public reset() {
    this.modelForm = deepCopy(this.model);
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      const filteredModel: ModelUpdate = filterUndefined(this.modelForm);
      await dispatchUpdateModel(this.$store, {
        id: this.model!.id,
        model: filteredModel,
      });
      this.$router.push('/main/models');
    }
  }

  get model() {
    return readOneModel(this.$store)(+this.$router.currentRoute.params.id);
  }
}
</script>
